import Foundation
import XCTest

@testable import FluidAudio

/// The persisted enrollment state must admit every state the streaming
/// updater actually produces. `spkcachePreds` is `nil` until the speaker
/// cache first overflows and `fifoPreds` is `nil` until the first chunk,
/// exactly as NeMo's `StreamingSortformerState` keeps them `None`; the old
/// validator compared `Int?` to `Int` and rejected all of them, so no bank
/// shorter than the cache could ever be exported (moonshot-mobile #6079).
final class SortformerEnrollmentSnapshotTests: XCTestCase {

    private let digest = "bank-digest"

    private var configuration: SortformerEnrollmentSnapshot.Configuration {
        let config = SortformerConfig.default
        return .init(
            chunkLen: config.chunkLen,
            chunkLeftContext: config.chunkLeftContext,
            chunkRightContext: config.chunkRightContext,
            fifoLen: config.fifoLen,
            spkcacheLen: config.spkcacheLen,
            spkcacheUpdatePeriod: config.spkcacheUpdatePeriod,
            spkcacheSilFramesPerSpk: config.spkcacheSilFramesPerSpk,
            numSpeakers: config.numSpeakers,
            preEncoderDims: config.preEncoderDims
        )
    }

    private func snapshot(
        spkcacheLength: Int,
        spkcachePreds: [Float]?,
        fifoLength: Int,
        fifoPreds: [Float]?,
        bankDigest: String? = nil,
        schemaVersion: Int = SortformerEnrollmentSnapshot.currentSchemaVersion
    ) -> SortformerEnrollmentSnapshot {
        let configuration = self.configuration
        return SortformerEnrollmentSnapshot(
            schemaVersion: schemaVersion,
            bankDigest: bankDigest ?? digest,
            configuration: configuration,
            spkcache: [Float](repeating: 0.1, count: spkcacheLength * configuration.preEncoderDims),
            spkcacheLength: spkcacheLength,
            spkcachePreds: spkcachePreds,
            fifo: [Float](repeating: 0.2, count: fifoLength * configuration.preEncoderDims),
            fifoLength: fifoLength,
            fifoPreds: fifoPreds,
            meanSilenceEmbedding: [Float](repeating: 0, count: configuration.preEncoderDims),
            silenceFrameCount: 0,
            speakers: [.init(slot: 0, name: "owner")]
        )
    }

    private func validate(_ snapshot: SortformerEnrollmentSnapshot) throws {
        try SortformerDiarizer.validateEnrollmentSnapshot(
            snapshot, against: configuration, expectedBankDigest: digest)
    }

    // MARK: - the states the updater produces

    func testShortBankWithNoPredictionsIsValid() throws {
        // One 15 s enrollment clip: a few frames popped into the cache,
        // nothing compressed yet, so both prediction arrays are still nil.
        let configuration = self.configuration
        try validate(snapshot(
            spkcacheLength: 31, spkcachePreds: nil,
            fifoLength: 0, fifoPreds: nil))
        XCTAssertLessThanOrEqual(31, configuration.spkcacheLen)
    }

    func testEmptyCacheWithNoPredictionsIsValid() throws {
        try validate(snapshot(
            spkcacheLength: 0, spkcachePreds: nil,
            fifoLength: 0, fifoPreds: nil))
    }

    func testFilledFIFOWithPredictionsAndUncompressedCacheIsValid() throws {
        let speakers = configuration.numSpeakers
        try validate(snapshot(
            spkcacheLength: 62, spkcachePreds: nil,
            fifoLength: 12, fifoPreds: [Float](repeating: 0.5, count: 12 * speakers)))
    }

    func testCompressedCacheCarriesPredictions() throws {
        let configuration = self.configuration
        let speakers = configuration.numSpeakers
        try validate(snapshot(
            spkcacheLength: configuration.spkcacheLen,
            spkcachePreds: [Float](repeating: 0.5, count: configuration.spkcacheLen * speakers),
            fifoLength: 3, fifoPreds: [Float](repeating: 0.5, count: 3 * speakers)))
    }

    // MARK: - the states that stay malformed

    func testFIFOFramesWithoutPredictionsAreMalformed() {
        XCTAssertThrowsError(try validate(snapshot(
            spkcacheLength: 0, spkcachePreds: nil,
            fifoLength: 6, fifoPreds: nil))
        ) { error in
            XCTAssertEqual(error as? SortformerEnrollmentSnapshotError, .malformedState)
        }
    }

    func testOverflowedCacheWithoutPredictionsIsMalformed() {
        let configuration = self.configuration
        XCTAssertThrowsError(try validate(snapshot(
            spkcacheLength: configuration.spkcacheLen + 1, spkcachePreds: nil,
            fifoLength: 0, fifoPreds: nil))
        ) { error in
            XCTAssertEqual(error as? SortformerEnrollmentSnapshotError, .malformedState)
        }
    }

    func testPredictionCountMismatchIsMalformed() {
        let speakers = configuration.numSpeakers
        XCTAssertThrowsError(try validate(snapshot(
            spkcacheLength: 10,
            spkcachePreds: [Float](repeating: 0.5, count: 9 * speakers),
            fifoLength: 0, fifoPreds: nil))
        ) { error in
            XCTAssertEqual(error as? SortformerEnrollmentSnapshotError, .malformedState)
        }
    }

    func testBankDigestMismatchIsNamed() {
        XCTAssertThrowsError(try validate(snapshot(
            spkcacheLength: 0, spkcachePreds: nil,
            fifoLength: 0, fifoPreds: nil,
            bankDigest: "another-bank"))
        ) { error in
            XCTAssertEqual(error as? SortformerEnrollmentSnapshotError, .bankDigestMismatch)
        }
    }

    func testUnsupportedSchemaIsNamed() {
        XCTAssertThrowsError(try validate(snapshot(
            spkcacheLength: 0, spkcachePreds: nil,
            fifoLength: 0, fifoPreds: nil,
            schemaVersion: SortformerEnrollmentSnapshot.currentSchemaVersion + 1))
        ) { error in
            XCTAssertEqual(error as? SortformerEnrollmentSnapshotError, .unsupportedSchema)
        }
    }
}
